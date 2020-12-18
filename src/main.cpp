
#include "imgui_setup.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _USE_MATH_DEFINES
#include <cmath>
#include "../depends/stb/stb_image.h"
#include "../depends/stb/stb_image_write.h"
#include <bits/stdc++.h>
#include <assert.h>
#include <math.h>
#include <random>
#include <ctime>
#include <functional>

#define RENDER_BATCH_COLUMNS 16 // Number of columns to render in a single go. Increase to gain some display/render speed!

int screen_width = 512, screen_height = 512; // This is window size, used to display scaled raytraced image.
int image_width = 256, image_height = 256; // This is raytraced image size. Change it if needed.
GLuint texImage;

using namespace std;

static default_random_engine generator(time(nullptr));
static uniform_real_distribution<float> distribution(0,1);
static auto myrand = std::bind(distribution,generator);

class Plane;

/***********************************************************************Vector3d********************************************************************************/
class Vector3D
{
  public:
    Vector3D() {}
  	Vector3D(double e0, double e1, double e2)
    {
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
    }
    Vector3D(const Vector3D &v)
    {
     	*this = v;
    }

    double X() const{ return e[0];}
    double Y() const{ return e[1];}
    double Z() const{ return e[2];}

    void X(double x) {e[0] = x;}
    void Y(double y) {e[1] = y;}
    void Z(double z) {e[2] = z;}

  	double getCoordinateByIndex(int index)
    {
       	return e[index];
    }

  	void setCoordinateByIndex(int index, double var)
    {
		e[index]=var;
    }

    //define operators
    const Vector3D& operator+() const;
	Vector3D operator-() const;
	double operator[](int i) const {return e[i];}
	double& operator[](int i) {return e[i];}

	friend bool operator==(const Vector3D& v1, const Vector3D& v2);
	friend bool operator!=(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D operator+(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D operator-(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D operator/(const Vector3D& v, double scalar);
	friend Vector3D operator*(const Vector3D& v, double scalar);
	friend Vector3D operator*(double scalar, const Vector3D& v);
	Vector3D& operator+=(const Vector3D &v);
	Vector3D& operator-=(const Vector3D &v);
	Vector3D& operator*=(double scalar);
	Vector3D& operator/=(double scalar);

	//Vector3D functions
	double length() const;
	double squaredlength() const;
	void normalize();
	friend Vector3D unitVector(const Vector3D& v);
	friend Vector3D crossProduct(const Vector3D& v1, const Vector3D& v2);
	friend double dotProduct(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D vecProduct(const Vector3D& v1, const Vector3D& v2);
	friend double tripleProduct(const Vector3D& v1,const Vector3D& v2,const Vector3D& v3);
	friend double calc_determinant(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3);

	//data member
    double e[3];
};

const Vector3D& Vector3D::operator+() const
{return *this;}

Vector3D Vector3D::operator-() const
{return Vector3D(-e[0], -e[1], -e[2]);}

bool operator==(const Vector3D& v1, const Vector3D& v2)
{
	if (v1.e[0] != v2.e[0]) return false;
	if (v1.e[1] != v2.e[1]) return false;
	if (v1.e[2] != v2.e[2]) return false;
	return true;
}

bool operator!=(const Vector3D& v1, const Vector3D& v2)
{
	return !(v1==v2);
}

Vector3D operator+(const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}

Vector3D operator-(const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);
}

Vector3D operator/(const Vector3D& v, double scalar)
{
	return Vector3D(v.e[0]/scalar, v.e[1]/scalar, v.e[2]/scalar);
}

Vector3D operator*(const Vector3D& v, double scalar)
{
	return Vector3D(v.e[0]*scalar, v.e[1]*scalar, v.e[2]*scalar);
}

Vector3D operator*(double scalar, const Vector3D& v)
{
	return Vector3D(v.e[0]*scalar, v.e[1]*scalar, v.e[2]*scalar);
}

Vector3D& Vector3D::operator+=(const Vector3D &v)
{
	e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2];
	return *this;
}

Vector3D& Vector3D::operator-=(const Vector3D &v)
{
	e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2];
	return *this;
}

Vector3D& Vector3D::operator*=(double scalar)
{
	e[0] *= scalar; e[1] *= scalar; e[2] *= scalar;
	return *this;
}

Vector3D& Vector3D::operator/=(double scalar)
{
	assert(scalar != 0);
	float inv = 1.f/scalar;
	e[0] *= inv; e[1] *= inv; e[2] *= inv;
	return *this;
}

double Vector3D::length() const
{ return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }

double Vector3D::squaredlength() const
{ return (e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }

void Vector3D::normalize()
{ *this = *this / (*this).length();}

Vector3D unitVector(const Vector3D& v)
{
	double length  = v.length();
	return v / length;
}

Vector3D crossProduct(const Vector3D& v1, const Vector3D& v2)
{
	Vector3D tmp;
	tmp.e[0] = v1.Y() * v2.Z() - v1.Z() * v2.Y();
	tmp.e[1] = v1.Z() * v2.X() - v1.X() * v2.Z();
	tmp.e[2] = v1.X() * v2.Y() - v1.Y() * v2.X();
	return tmp;
}

Vector3D vecProduct(const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D(v1.X()*v2.X(), v1.Y()*v2.Y(), v1.Z()*v2.Z());
}

double dotProduct(const Vector3D& v1, const Vector3D& v2)
{ return v1.X()*v2.X() + v1.Y()*v2.Y() + v1.Z()*v2.Z(); }

double tripleProduct(const Vector3D& v1,const Vector3D& v2,const Vector3D& v3)
{
	return dotProduct(( crossProduct(v1, v2)), v3);
}

double determinant(const Vector3D& v1,const Vector3D& v2,const Vector3D& v3)
{
  double temp = v1.X()*(v2.Y()*v3.Z()-v3.Y()*v2.Z())
               -v2.X()*(v1.Y()*v3.Z()-v3.Y()*v1.Z())
               +v3.X()*(v1.Y()*v2.Z()-v2.Y()*v1.Z());
  return temp;
}




class Photon
{
	public:
		Vector3D position;	//Photon Position
		Vector3D power;		//Photon Power
		unsigned char phi, theta;	//Incoming Direction
		short planeFlag;	//KD-Tree's Splitting Plane

		//Constructor
		Photon(Vector3D pos, Vector3D dir, Vector3D pow)
		{
			position = pos;
			power = pow;

			int theta_val = int(acos(dir.Z()) * 256.0 / M_PI);
			if(theta_val > 255)
			{
				theta = theta_val;
			}
			else
			{
				theta = (unsigned char)theta_val;
			}

			int phi_val = int(atan2(dir.Y(), dir.X()) * (256.0 / 2.0 * M_PI));
			if(phi_val > 255)
			{
				phi = 255;
			}
			else if(phi_val < 0)
			{
				phi = (unsigned char)(phi + 256);
			}
			else
			{
				phi = (unsigned char)phi;
			}

		}
};

class NearestPhotons
{
	public:
		int max;
		int found;
		int gotHeap;
		Vector3D pos;
		float *dist;
		const Photon **index;
};

#define swap(photon, i, j) { Photon *tempPhoton = photon[i]; photon[i] = photon[j]; photon[j] = tempPhoton; }

class PhotonMap
{
	public:
		//Variables
		Photon *photons;
		float sinTheta[256];
		float sinPhi[256];
		float cosTheta[256];
		float cosPhi[256];
		int storedPhotons;
		int halfStoredPhotons;
		int maxPhotons;
		int prevScale;
		Vector3D bboxMin;
		Vector3D bboxMax;

		//Constructor
		PhotonMap(int maxPhotonsCount)
		{
			storedPhotons = 0;
			prevScale = 1;
			maxPhotons = maxPhotonsCount;

			photons = (Photon*)malloc(sizeof(Photon) * (maxPhotons + 1));
			if(photons == NULL)
			{
				cout<<"Unable to allocate required Memory for Photons!!!\n"<<endl;
				exit(0);
			}

			for(int i = 0;i < 256;i++)
			{
				double angle = double(i) * (1.0 / 256.0) * M_PI;
				sinTheta[i] = sin(angle);
				cosTheta[i] = cos(angle);
				sinPhi[i] = sin(2.0 * angle);
				cosPhi[i] = cos(2.0 * angle);
			}

			bboxMin = Vector3D(1e5f, 1e5f, 1e5f);
			bboxMax = Vector3D(-1e5f, -1e5f, -1e5f);
		}

		void photonDir(float *dir, const Photon *p) const
		{
			dir[0] = sinTheta[p->theta] * cosPhi[p->phi];
			dir[1] = sinTheta[p->theta] * sinPhi[p->phi];
			dir[2] = cosTheta[p->theta];
		}

		void locatePhotons(NearestPhotons *const np, const int index) const
		{
			Photon *p = &photons[index];
			double dist, distSum;

			if(index < halfStoredPhotons)
			{
				dist = np->pos.getCoordinateByIndex(p->planeFlag) - p->position.getCoordinateByIndex(p->planeFlag);

				// Search in right planeFlag if dist > 0
				if(dist > 0.0)
				{
					locatePhotons(np, 2 * index + 1);
					if(dist * dist < np->dist[0])
					{
						locatePhotons(np, 2 * index);
					}
				}
				// Otherwise search in left planeFlag if dist <= 0
				else
				{
					locatePhotons(np, 2 * index);
					if(dist * dist < np->dist[0])
					{
						locatePhotons(np, 2 * index + 1);
					}
				}
			}

			// Compute squared distance between current photon and nearestPhoton
			dist = p->position.X() - np->pos.X();
			distSum = dist * dist;
			dist = p->position.Y() - np->pos.Y();
			distSum += dist * dist;
			dist = p->position.Z() - np->pos.Z();
			distSum += dist * dist;

			if(distSum < np->dist[0])
			{
				// Nearest Photon is found. Now, insert it in candidate list
				if(np->found < np->max)
				{
					// Heap is not full, use array
					np->found++;
					np->dist[np->found] = distSum;
					np->index[np->found] = p;
				}
				else
				{
					int temp, parent;
					if(np->gotHeap == 0)
					{
						// Build heap
						double dist1;
						const Photon *tempPhoton;
						int halfFound = np->found >> 1;
						for(int i = halfFound;i >= 1;i--)
						{
							parent = i;
							tempPhoton = np->index[i];
							dist1 = np->dist[i];
							while(parent <= halfFound)
							{
								temp = parent + parent;
								if(temp < np->found && np->dist[temp] < np->dist[temp+1])
								{
									temp++;
								}
								if(dist1 >= np->dist[temp])
								{
									break;
								}
								np->dist[parent] = np->dist[temp];
								np->index[parent] = np->index[temp];
								parent = temp;
							}
						    np->dist[parent] = dist1;
						    np->index[parent] = tempPhoton;
						}
						np->gotHeap = 1;
					}

					/*
					* Insert new Photon into max heap
					* delete largest element, insert new and reorder heap
					*/

					parent = 1; temp = 2;
					while(temp <= np->found)
					{
						if(temp <= np->found && np->dist[temp] < np->dist[temp+1])
						{
							temp++;
						}
						if(distSum > np->dist[temp])
						{
							break;
						}
						np->dist[parent] = np->dist[temp];
						np->index[parent] = np->index[temp];
						parent = temp;
						temp += temp;
					}
					np->index[parent] = p;
					np->dist[parent] = distSum;
					np->dist[0] = np->dist[1];
				}
			}
		}

		void irradianceEstimate(Vector3D *irrad, Vector3D pos, Vector3D normal, const float maxDist, const int nPhotons)
		{
			irrad->X(0);
			irrad->Y(0);
			irrad->Z(0);
			NearestPhotons np;
			np.dist = (float*)alloca(sizeof(float) * (nPhotons + 1));
			np.index = (const Photon**)alloca(sizeof(Photon*) * (nPhotons + 1));
			np.pos.X(pos.X());
			np.pos.Y(pos.Y());
			np.pos.Z(pos.Z());
			np.max = nPhotons;
			np.found = np.gotHeap = 0;
			np.dist[0] = maxDist * maxDist;

			// Locate the nearest Photons
			locatePhotons(&np, 1);

			float pdir[3];

			//Total radiance from all the photons in photon map
			for (int i = 1; i <= np.found; i++)
			{
				const Photon *p = np.index[i];
		        photonDir(pdir,p);
		        if((pdir[0]*normal.X()+pdir[1]*normal.Y()+pdir[2]*normal.Z())<0.0f)
		        {
					irrad->X(irrad->X() + p->power.X());
					irrad->Y(irrad->Y() + p->power.Y());
					irrad->Z(irrad->Z() + p->power.Z());
		        }
			}
		}


		void store(Photon* p)
		{
			if(storedPhotons > maxPhotons)
			{
				return;
			}
			storedPhotons++;
			photons[storedPhotons] = *p;
			Photon *treeNode = &photons[storedPhotons];
			treeNode = p;
			if(treeNode->position.X() < bboxMin.X())
			{
				bboxMin.X(treeNode->position.X());
			}
			if(treeNode->position.Y() < bboxMin.Y())
			{
				bboxMin.Y(treeNode->position.Y());
			}
			if(treeNode->position.Z() < bboxMin.Z())
			{
				bboxMin.Z(treeNode->position.Z());
			}

			if(treeNode->position.X() > bboxMax.X())
			{
				bboxMax.X(treeNode->position.X());
			}
			if(treeNode->position.Y() > bboxMax.Y())
			{
				bboxMax.Y(treeNode->position.Y());
			}
			if(treeNode->position.Z() > bboxMax.Z())
			{
				bboxMax.Z(treeNode->position.Z());
			}
		}

		void printPhotons()
		{
			for(int i = 0;i < maxPhotons;i++)
			{
				Photon * treeNode = &photons[i];
				cout<<photons[i].position.X()<<" "<<photons[i].position.Y()<<" "<<photons[i].position.Z()<<endl;
			}
		}

		void scalePhotonPower(const double powerScale)
		{
			for(int i = prevScale;i <= storedPhotons;i++)
			{
				photons[i].power[0] *= powerScale;
				photons[i].power[1] *= powerScale;
				photons[i].power[2] *= powerScale;
			}
			prevScale = storedPhotons;
		}


		void medianSplit(Photon **p, const int start, const int end, const int median, const int index)
		{
			int left = start, right = end;
			while(right > left)
			{
				const float v = p[right]->position.getCoordinateByIndex(index);
				int i = left - 1, j = right;
				for(;;)
				{
					while(p[++i]->position.getCoordinateByIndex(index) < v && i < right);
					while(p[--j]->position.getCoordinateByIndex(index) > v && j > left);
					if(i >= j)
					{
						break;
					}
					swap(p, i, j);
				}
				swap(p, i, right);
				if(i >= median)
				{
					right = i -1;
				}
				if(i <= median)
				{
					left = i + 1;
				}
			}
		}

		void balanceSegment(Photon **pBal, Photon **pOrg, const int ind, const int start, const int end)
		{
			// Calculate new median
			int median = 1;
			while((4 * median)  <= (end - start +1))
			{
				median += median;
			}
			if((3 * median) <= (end - start +1))
			{
				median += median;
				median += start - 1;
			}
			else
			{
				median = end - median + 1;
			}

			// Find the index at which we need to split
			int index = 2;
			if((bboxMax.X() - bboxMin.X()) > (bboxMax.Y() - bboxMin.Y()) && (bboxMax.X() - bboxMin.X()) > (bboxMax.Z() - bboxMin.Z()))
			{
				index = 0;
			}
			else if((bboxMax.Y() - bboxMin.Y()) > (bboxMax.Z() - bboxMin.Z()))
			{
				index = 1;
			}

			// Split the photons with median
			medianSplit(pOrg, start, end, median, index);
			pBal[ind] = pOrg[median];
			pBal[ind]->planeFlag = index;

			// Balance left and right sub arrays recursively
			if(median > start)
			{
				// Balance left sub array
				if(start < median - 1)
				{
					const float tmp = bboxMax.getCoordinateByIndex(index);
					bboxMax.setCoordinateByIndex(index, pBal[ind]->position.getCoordinateByIndex(index));
					balanceSegment(pBal, pOrg, 2 * ind, start, median - 1);
					bboxMax.setCoordinateByIndex(index, tmp);
				}
				else
				{
					pBal[2 * ind] = pOrg[start];
				}
			}
			if(median < end)
			{
				// Balance right sub array
				if(median + 1 < end)
				{
					const float tmp = bboxMin.getCoordinateByIndex(index);
					bboxMin.setCoordinateByIndex(index, pBal[ind]->position.getCoordinateByIndex(index));
					balanceSegment(pBal, pOrg, 2 * ind + 1, median + 1, end);
					bboxMin.setCoordinateByIndex(index, tmp);
				}
				else
				{
					pBal[2 * ind + 1] = pOrg[end];
				}
			}
		}

		void balance()
		{
			if(storedPhotons > 1)
			{
				Photon **pa1 = (Photon**)malloc(sizeof(Photon*) * (storedPhotons + 1));
				Photon **pa2 = (Photon**)malloc(sizeof(Photon*) * (storedPhotons + 1));
				for (int i = 0; i < storedPhotons + 1; i++)
				{
			    	pa2[i] = &photons[i];
				}
				balanceSegment(pa1, pa2, 1, 1, storedPhotons);
				free(pa2);

				// Reorganize balanced kd-tree to make a heap
				int d, j = 1, foo = 1;
				Photon fooPhoton = photons[j];

				for (int i = 1; i < storedPhotons + 1; i++)
				{
				    d = pa1[j] - photons;
				    pa1[j] = NULL;
				    if( d != foo)
				    {
				    	photons[j] = photons[d];
				    }
				    else
			    	{
						photons[j] = fooPhoton;

						if(i < storedPhotons)
						{
							for (; foo < storedPhotons + 1; foo++)
							{
								if(pa1[foo] != NULL)
								{
									break;
								}
							}
							fooPhoton = photons[foo];
							j = foo;
						}
						continue;
					}
					j = d;
				}
				free(pa1);
			}
			halfStoredPhotons = storedPhotons / 2 - 1;
		}
};

/*--------------------------------------------------------------------------Sphere---------------------------------------------------------------------*/
class Sphere
{
	public:
	    Vector3D position;
	    double radius;
	    Vector3D color;
  		int opProp;		//0 for Diffuse; 1 for Specular; 2 for Refract

	    //Constructors
	    Sphere()
	    {
			radius = 0;
			color = Vector3D(1.0, 1.0, 1.0);
	    }
	    Sphere(Vector3D pos, double r, int o, Vector3D c)
	    {
			position = pos;
			radius = r;
			color = c;
		    opProp = o;
	    }

	    double intersect(Vector3D origin, Vector3D direction)
	    {
	        Vector3D temp = Vector3D(position.X() - origin.X(), position.Y() - origin.Y(), position.Z() - origin.Z());
	        double a = dotProduct(direction,direction);
	        double b = -2.0 * dotProduct(temp, direction);
	        double c = dotProduct(temp, temp) - radius * radius;
	        double d = b*b - 4*a*c;
	        if(c < 0)
	        {
				return 1.0e6;
	        }
	        if(d > 0.0)
	        {
				return (-b - sqrt(d)) / (2.0 * a);
	        }
	        return 1.0e6;
	    }
	    Vector3D normal(Vector3D incident)
	    {
	        Vector3D norm = Vector3D(incident.X() - position.X(), incident.Y() - position.Y(), incident.Z() - position.Z());
	        norm.normalize();
	        return norm;
	    }
};
/*-----------------------------------------------------------------IntersectObject------------------------------------------------------------------------*/
class intersectObject
{
	public:
		int idx;	//Object Id: which object no it is
		int type;	//Object Type: 0 for sphere; 1 for plane; etc
		double distance;	//Dist between photon origin and intersection point

		//Constructor
		intersectObject()
		{
			idx = -1;
			type = -1;
			distance = 1.0e6;
		}
};
/*----------------------------------------------------------------------PhotonRay----------------------------------------------------------------------*/
class PhotonRay
{
	public:
		Vector3D origin;
		Vector3D direction;
		Vector3D color;

	  //Constructor
		PhotonRay()
		{
			origin = Vector3D(0.0, 0.0, 0.0);
			direction = Vector3D(0.0, 0.0, 0.0);
		}

		PhotonRay(Vector3D org)
		{
			origin = org;
			direction = Vector3D(0.0, 0.0, 0.0);
		}

		void randomDirection(double var)
		{
			direction.X((double)rand() * 2 * var / RAND_MAX - var);
			direction.Y((double)rand() * 2 * var / RAND_MAX - var);
			direction.Z((double)rand() * 2 * var / RAND_MAX - var);
			direction.normalize();
		}

		void reflect(Vector3D incident, Vector3D normal)
		{
			double reflectance = 2 * dotProduct(direction, normal);
			Vector3D newDirection = Vector3D(direction.X() - normal.X() * reflectance, direction.Y() - normal.Y() * reflectance, direction.Z() - normal.Z() * reflectance);
		  	newDirection.normalize();
		  	direction = newDirection;
		  	origin = incident;
		}

		void refract(Vector3D incident, Vector3D normal, double &ref)
		{
			double n1 = 1.0;
			double n2 = 1.8;
			double incidentCos  = dotProduct(direction, normal);

			if(incidentCos > 0)
			{
				double tempN = n1;
				n1 = n2;
				n2 = tempN;
			}

			double n  = n1 / n2;
			double refractedSin = n * n * (1.0 - incidentCos * incidentCos);
			double refractedCos = sqrt(1.0 - refractedSin);

			Vector3D newDirection = (direction * n) + normal * (n * incidentCos - refractedCos);
			newDirection.normalize();
			origin = incident;
			direction = newDirection;
		}

		void diffuse(Vector3D incident)
		{
			Vector3D intersect(origin-incident);
			intersect.normalize();
			PhotonRay newRay = PhotonRay(incident);
			newRay.randomDirection(1.0);
			origin = incident;
			direction = newRay.direction;
		}

		intersectObject* tracePhotonRay(vector<Plane*>, vector<Sphere*>);
};
/*---------------------------------------------------------------------Plane-------------------------------------------------------------------------*/
class Plane
{
	public:
		int axis;	//0 for x-axis; 1 for y-axis; 2 for z-axis
		double position;	//distance from origin in that axis
		Vector3D color;
		int opProp;  //0 for Diffuse; 1 for Specular; 2 for Refract

		//Constructor
		Plane(int ax, double pos, float r, float g, float b)
		{
			axis = ax;
			position = pos;
			color = Vector3D(r, g , b);
		    opProp = 0;
		}

		double intersect(Vector3D origin, Vector3D direction)
		{
		  	if(axis == 0)
		  	{
		  	 	if(direction.X() != 0.0)
		  	 	{
		    		return  (position - origin.X()) / direction.X();
		  	 	}
		  	}
		  	else if(axis == 1)
		  	{
		  		if(direction.Y() != 0.0)
		  		{
		  			return  (position - origin.Y()) / direction.Y();
		  		}
		  	}
		  	else if(axis == 2)
		  	{
		  		if(direction.Z() != 0.0)
		  		{
		  			return  (position - origin.Z()) / direction.Z();
		  		}
		  	}
		  	return 1.0e6;
		}
		Vector3D normal(Vector3D incident, PhotonRay pray)
		{
		  	Vector3D norm = Vector3D(0.0, 0.0, 0.0);
		  	if(axis == 0)
		  	{
		  		norm.X(pray.origin.X() - position);
		  	}
		  	if(axis == 1)
		  	{
		  		norm.Y(pray.origin.Y() - position);
		  	}
		  	if(axis == 2)
		  	{
		  		norm.Z(pray.origin.Z() - position);
		  	}
		  	norm.normalize();
		  	return norm;
		}
		Vector3D getNormal()
		{
			if(axis == 0)
		  	{
		  		return Vector3D(1.0, 0.0, 0.0);
		  	}
		  	if(axis == 1)
		  	{
		  		return Vector3D(0.0, 1.0, 0.0);
		  	}
		  	if(axis == 2)
		  	{
		  		return Vector3D(0.0, 0.0, 1.0);
		  	}

		}
};

intersectObject* PhotonRay::tracePhotonRay(vector<Plane*> planes, vector<Sphere*> spheres)
{
	intersectObject *object = new intersectObject();

	// Check intersection of photon ray with all the planes
	for(int i = 0;i < planes.size();i++)
	{
		double dist = planes[i]->intersect(origin, direction);
		// cout<<dist<<endl;
		if(dist < object->distance && dist > 1.0e-5)
		{
			object->idx = i;
			object->type = 1;
			object->distance = dist;
		}
	}

  // Check intersection of photon ray with all the spheres
	for(int i = 0;i < spheres.size();i++)
	{
		double dist = spheres[i]->intersect(origin, direction);
		if(dist < object->distance and dist > 1.0e-5)
		{
			object->idx = i;
			object->type = 0;
			object->distance = dist;
		}
	}
	return object;
}

/*---------------------------------------------------------------------Light--------------------------------------------------------------------------*/
class Light
{
	public:
		const int numPhotons = 100000;
		Vector3D position;
		double power;
		Vector3D color;

		//Constructor
		Light(Vector3D pos, double pow, Vector3D col)
		{
			position = pos;
			power = pow;
			color = col;
		}
};
/*----------------------------------------------------------------------World-------------------------------------------------------------------------*/
class World
{
	public:
		vector<Plane*> planes;		//List of all plane objects
		vector<Sphere*> spheres;	//List of all sphere objects
		Light* light;				//Light Source
		PhotonMap *cMap;
  		PhotonMap *gMap;

		//Constructor
		World()
		{
			//Generate Planes
			Plane* left = new Plane(0, 1.5, 0, 1, 1);
			Plane* right = new Plane(0, -1.5, 0, 1, 0);
			Plane* top = new Plane(1, 1.5, 0, 0, 1);
			Plane* bottom = new Plane(1, -1.5, 0, 1, 1);
			Plane* back = new Plane(2, -5.0, 1, 1, 0);
			Plane* behind = new Plane(2, 5.1, 1, 0, 1);
			planes.push_back(left);
			planes.push_back(right);
			planes.push_back(top);
			planes.push_back(bottom);
			planes.push_back(back);
			planes.push_back(behind);
			//Generate Spheres
			Sphere* s1 = new Sphere(Vector3D(-1.0, -1.0, -3.0), 0.5, 1, Vector3D(1.0, 1.0, 1.0));
			Sphere* s2 = new Sphere(Vector3D(0.5, -0.7, -2.5), 0.8, 2, Vector3D(1.0, 0.0, 0.0));
			spheres.push_back(s1);
			spheres.push_back(s2);

			cMap = new PhotonMap(600000);
			gMap = new PhotonMap(600000);
			light = new Light(Vector3D(0.0, 1.49, -2.0), 100000, Vector3D(1.0, 1.0, 1.0));
		}

		// Generate photons and beam them in random directions and store them in the photon map
		void generatePhotons()
		{
			srand(0);
			float photonPower = 1;
			for(int i = 0;i < light->numPhotons;i++)
			{
				// cout<<i<<"\n";
				int numBounces = 1;
		        double refractive = 1.0;
				int flag1 = 0;
				int flag2 = 0;
				PhotonRay pRay(light->position);
				pRay.randomDirection(1.0);

				Vector3D objectColor(1.0, 1.0, 1.0);
				Vector3D colorIntensity(1.0, 1.0, 1.0);

				Vector3D normal;
				intersectObject *object = pRay.tracePhotonRay(planes, spheres);

		        if(fabs(pRay.origin.X()) > 1.5 || pRay.origin.Y() > 1.5)
		        {
					continue;
		        }

				while(object->idx != -1 and numBounces <= 8)
				{
			        Vector3D incidentRay = Vector3D(pRay.origin.X() + pRay.direction.X() * object->distance, pRay.origin.Y() + pRay.direction.Y() * object->distance, pRay.origin.Z() + pRay.direction.Z() * object->distance);

	          		if(incidentRay.Z() > 0 || incidentRay.Z() < -5 || fabs(incidentRay.Y()) > 1.5 || fabs(incidentRay.X()) > 1.5 )
					{
						break;
					}

			      	int objectIndex = object->idx;

			      	if(object->type == 0)
			      	{
			      		++flag1;
			      		normal = spheres[objectIndex]->normal(incidentRay);
			      		colorIntensity = vecProduct(objectColor , spheres[objectIndex]->color);
			      		objectColor = colorIntensity * (1.0 / sqrt((double)numBounces));
			      		Photon *p = new Photon(incidentRay, pRay.direction, Vector3D(objectColor.X() * photonPower, objectColor.Y() * photonPower, objectColor.Z() * photonPower));
			      		gMap->store(p);

		                if(spheres[objectIndex]->opProp == 1)
		                {
							pRay.reflect(incidentRay, normal);
		                }

		                else if(spheres[objectIndex]->opProp == 2)
		                {
							pRay.refract(incidentRay, normal, refractive);
		                }

			      	}

			      	else
			      	{
			      		normal = planes[objectIndex]->normal(incidentRay, pRay);
			      		float cosTheta = abs(dotProduct(normal, incidentRay) / (incidentRay.length() * normal.length()));
			      		colorIntensity = vecProduct(objectColor , planes[objectIndex]->color);
			      		objectColor = colorIntensity * (1.0 / sqrt((double)numBounces));
			      		Photon *p = new Photon(incidentRay, pRay.direction,Vector3D(objectColor.X() * photonPower, objectColor.Y() * photonPower, objectColor.Z() * photonPower));
			      		if(flag1 > 0 and flag2==0)
			      		{
			      			cMap->store(p);
			      			flag2 = 1;
			      		}
		                else
		                {
							gMap->store(p);
		                }
			      		pRay.diffuse(incidentRay);
			      	}

			      	object = pRay.tracePhotonRay(planes, spheres);
			      	++numBounces;
			    }
		    }
		    cout<<"Total No of Caustic Map Photons:"<<cMap->storedPhotons<<endl;
	        cout<<"Total No of Global Map Photons:"<<gMap->storedPhotons<<endl;
		    cMap->balance();
		    cMap->scalePhotonPower(1.0f/light->numPhotons);
	        gMap->balance();
	        gMap->scalePhotonPower(1.0f/light->numPhotons);
		}

		Vector3D irradiance(PhotonMap* pMap, const Vector3D& origin, const Vector3D& direction, const vector<Plane*> planes, const vector<Sphere*> spheres, int depth, int RI)
		{

		    double t;
		    Vector3D n;
		    Vector3D f;
		    Vector3D newOrigin;
		    int op;

		    intersectObject *object = new intersectObject();

		    for(int i = 0;i < planes.size();i++)
		    {
				double dist = planes[i]->intersect(origin, direction);
				if(dist < object->distance && dist > 1.0e-5)
				{
			        object->idx = i;
			        object->type = 1;
			        object->distance = dist;
			        newOrigin = origin + direction * dist;
			        t=dist;
			        n=planes[i]->getNormal();
			        f=planes[i]->color;
			        op=planes[i]->opProp;
				}
		    }

		    // Check intersection of photon ray with all the spheres
		    for(int i = 0;i < spheres.size();i++)
		    {
				double dist = spheres[i]->intersect(origin, direction);
				if(dist < object->distance and dist > 1.0e-5)
				{
			        object->idx = i;
			        object->type = 0;
			        object->distance = dist;
			        newOrigin = origin + direction * dist;
			        t=dist;
			        n=spheres[i]->normal(newOrigin);
			        f=spheres[i]->color;
			        op=spheres[i]->opProp;
				}
		    }

		    if(object->idx == -1)		//No Object Intersection
		    {
		      return Vector3D(0,0,0);
		    }

		    Vector3D nl = dotProduct(n, direction)<0?n:n*-1;

		    if(++depth>5)
		    {
				return Vector3D(0,0,0);
		    }

		    if(op == 0)
		    {
				Vector3D color;
				Vector3D pos(newOrigin.X(),newOrigin.Y(),newOrigin.Z());
				pMap->irradianceEstimate(&color,pos,n,0.1,100);
				return color;
		    }

		    else if(op == 1)
		    {
				return vecProduct(f,(irradiance(pMap, newOrigin, direction - n*2*dotProduct(n, direction), planes, spheres, depth, RI)));
		    }

		    else if(op == 2)
		    {
				Vector3D reflRay(direction-n*2*dotProduct(n, direction));     // Ideal dielectric REFRACTION
				bool into = dotProduct(n, nl)>0;                // Ray from outside going in?
				float nc=1, nt=RI, nnt=into?nc/nt:nt/nc, ddn=dotProduct(direction, nl), cos2t;
				if((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)
				{			// Total internal reflection
			        return vecProduct(f,(irradiance(pMap, newOrigin, reflRay, planes, spheres, depth, RI)));
				}

				Vector3D tdir = (direction*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t))));
				tdir.normalize();
				float a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:dotProduct(tdir,n));
				float Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re;
				return f + vecProduct(f,(irradiance(pMap, newOrigin, reflRay, planes, spheres, depth, RI)*Re + irradiance(pMap, newOrigin, tdir, planes, spheres, depth, RI)*Tr));
		    }

		    return Vector3D(0,0,0);
		}

		Vector3D raytrace(PhotonMap* cMap, PhotonMap* gMap, const Vector3D& origin, const Vector3D& direction, const vector<Plane*> planes, const vector<Sphere*> spheres,Light* light, int depth, int RI)
		{

		    double t;
		    Vector3D n;
		    Vector3D f;
		    Vector3D newOrigin;
		    int op;

		    intersectObject *object = new intersectObject();

		    for(int i = 0;i < planes.size();i++)
		    {
				double dist = planes[i]->intersect(origin, direction);
				// cout<<dist<<"\n";
				if(dist < object->distance && dist > 1.0e-5)
				{
					object->idx = i;
					object->type = 1;
					object->distance = dist;
					newOrigin = origin + direction * dist;
					t=dist;
					n=planes[i]->getNormal();
					if(i==3)
					{
						n=n*-1;
					}
					f=planes[i]->color;
					op=planes[i]->opProp;
				}
		    }

		    // Check intersection of photon ray with all the spheres
		    for(int i = 0;i < spheres.size();i++)
		    {
				double dist = spheres[i]->intersect(origin, direction);
				// cout<<dist<<"\n";
				if(dist < object->distance and dist > 1.0e-5)
				{
					object->idx = i;
					object->type = 0;
					object->distance = dist;
					newOrigin = origin + direction * dist;
					t=dist;
					n=spheres[i]->normal(newOrigin);
					f=spheres[i]->color;
					op=spheres[i]->opProp;
				}
		    }

		    if(object->idx == -1)		//No Object Intersection
		    {
				return Vector3D(0,0,0);
		    }

		    Vector3D nl = dotProduct(n, direction)<0?n:n*-1;

		    if(++depth>20)
		    {
				return Vector3D(0,0,0);
		    }

		    if(op == 0)
		    {
				//Caustic
				Vector3D color;
				Vector3D pos(newOrigin.X(),newOrigin.Y(),newOrigin.Z());
				cMap->irradianceEstimate(&color,pos,n,0.1,100);

				//Global

				int nsamps = 256;
				for (int i = 0; i<nsamps ; i++)
				{
			        double r1=2*M_PI*myrand(), r2=myrand(), r2s=sqrt(r2);
					Vector3D w=nl;
					Vector3D u=fabs(w.X())>.1?Vector3D(0,1,0):crossProduct(Vector3D(1,0,0),w);
					u.normalize();
					Vector3D v=crossProduct(w,u);
					Vector3D d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2));
					d.normalize();
			        color = color + irradiance(gMap, newOrigin, d, planes, spheres, 0, RI)*(1.f/nsamps);
				}

				//Direct

				Vector3D d1=light->position - newOrigin;
				double tLight = d1.length();
				d1.normalize();


				double t1;
				Vector3D n1;

				intersectObject *object1 = new intersectObject();

				for(int i = 0;i < planes.size();i++)
				{
			        double dist1 = planes[i]->intersect(newOrigin, d1);
			        if(dist1 < object1->distance && dist1 > 1.0e-5)
			        {
						object1->idx = i;
						object1->type = 1;
						object1->distance = dist1;
						t1=dist1;
						n1=planes[i]->getNormal();
						if(i==3)
						{
							n1=n1*-1;
						}
			        }
				}

				// Check intersection of photon ray with all the spheres
				for(int i = 0;i < spheres.size();i++)
				{
			        double dist1 = spheres[i]->intersect(newOrigin, d1);
			        if(dist1 < object1->distance and dist1 > 1.0e-5)
			        {
						object1->idx = i;
						object1->type = 0;
						object1->distance = dist1;
						t1=dist1;
						n1=spheres[i]->normal(newOrigin + d1 * dist1);
			        }
				}

				if(object1->idx !=-1 || t1>tLight)
				{
					color = color + vecProduct(f,(light->color*dotProduct(d1,n1)));
				}

				return color;
		    }

		    else if(op == 1)
		    {
				return vecProduct(f,(raytrace(cMap, gMap, newOrigin, direction - n*2*dotProduct(n, direction), planes, spheres, light, depth, RI)));
		    }

		    else
		    {
				Vector3D reflRay(direction-n*2*dotProduct(n, direction));     // Ideal dielectric REFRACTION
				bool into = dotProduct(n, nl)>0;                // Ray from outside going in?
				float nc=1, nt=RI, nnt=into?nc/nt:nt/nc, ddn=dotProduct(direction, nl), cos2t;
				if((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)		// Total internal reflection
				{			
			        return vecProduct(f,(raytrace(cMap, gMap, newOrigin, reflRay, planes, spheres, light, depth, RI)));
				}

				Vector3D tdir = (direction*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t))));
				tdir.normalize();
				float a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:dotProduct(tdir,n));
				float Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re;
				return f + vecProduct(f,(raytrace(cMap, gMap, newOrigin, reflRay, planes, spheres, light, depth, RI)*Re + raytrace(cMap, gMap, newOrigin, tdir, planes, spheres, light, depth, RI)*Tr));

		    }
		}
};


//-------------------------------------------------------------xxxx-------------------------------------------------------------

class Camera
{
private:
	Vector3D position;
	Vector3D target; //Look-at point
	Vector3D up;

	Vector3D line_of_sight;
	Vector3D u, v, w; //Camera basis vectors

	unsigned char *bitmap;
	int width, height;
	float fovy;// expressed in degrees: FOV-Y; angular extent of the height of the image plane
	float focalDistance; //Distance from camera center to the image plane
	float focalWidth, focalHeight;//width and height of focal plane
	float aspect;

public:
	Camera(const Vector3D& _pos, const Vector3D& _target, const Vector3D& _up, float _fovy, int _width, int _height)
	{
		position = _pos;
		target = _target;
		up = _up;
		fovy = _fovy;
		width = _width;
		height = _height;
		up.normalize();

		line_of_sight = target - position;

		//Calculate the camera basis vectors
		//Camera looks down the -w axis
		w = -line_of_sight;
		w.normalize();
		u = crossProduct(up, w);
		u.normalize();
		v = crossProduct(w, u);
		v.normalize();
		bitmap  = new unsigned char[width * height * 3]; //RGB
		for (std::size_t i = 0; i < 3*width*height; ++i) {
			bitmap[i] = 0;
		}
		focalHeight = 1.0; //Let's keep this fixed to 1.0
		aspect = float(width)/float(height);
		focalWidth = focalHeight * aspect; //Height * Aspect ratio
		focalDistance = focalHeight/(2.0 * tan(fovy * M_PI/(180.0 * 2.0))); //More the fovy, close is focal plane
	}
	~Camera()
	{
		delete []bitmap;
	}
	const Vector3D get_ray_direction(const int i, const int j) const
	{
		Vector3D dir(0.0, 0.0, 0.0);
		dir += -w * focalDistance;
		float xw = aspect*(i - width/2.0 + 0.5)/width;
		float yw = (j - height/2.0 + 0.5)/height;
		dir += u * xw;
		dir += v * yw;

		dir.normalize();
		return dir;
	}
	const Vector3D& get_position() const
	{
		return position;
	}
	void drawPixel(int i, int j, Vector3D c)
	{
		int index = (i + j*width)*3;
		bitmap[index + 0] = 255 * c.X();
		bitmap[index + 1] = 255 * c.Y();
		bitmap[index + 2] = 255 * c.Z();
	}

	unsigned char * getBitmap()
	{
		return bitmap;
	}
	int getWidth()
	{
		return width;
	}
	int getHeight()
	{
		return height;
	}

};


class RenderEngine
{
public:
	Camera *camera;
	RenderEngine(Camera *_camera): camera(_camera) {}

	bool renderLoop(World *world)
	{
		static int i = 0;
		for(int j = 0; j<camera->getHeight(); j++)
		{
			Vector3D ray_dir = camera->get_ray_direction(i, j);
			Vector3D origin = camera->get_position();

			Vector3D color = world->raytrace(world->cMap, world->gMap, origin, ray_dir, world->planes, world->spheres, world->light, 0, 1.8);
			// cout<<"i:"<<i<<" j:"<<j<<"   "<<color.X()<<" "<<color.Y()<<" "<<color.Z()<<"\n";
			camera->drawPixel(i, j, color);
		}

		if(++i == camera->getWidth())
		{
			i = 0;
			return true;
		}
		return false;
	}
};

int main(int, char**)
{
	Camera *camera;
	RenderEngine *engine;
    GLFWwindow *window = setupWindow(screen_width, screen_height);

    ImVec4 clearColor = ImVec4(1.0f, 1.0f, 1.0f, 1.00f);

    // Setup raytracer camera. This is used to spawn rays.
    Vector3D camera_position(0, 0, 3);
    Vector3D camera_target(0, 0, 0); //Looking down -Z axis
    Vector3D camera_up(0, 1, 0);
    float camera_fovy =  45;
    camera = new Camera(camera_position, camera_target, camera_up, camera_fovy, image_width, image_height);

    World* pWorld=new World();
    pWorld->generatePhotons();
    
    engine = new RenderEngine(camera);

    glGenTextures(1, &texImage);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texImage);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, camera->getBitmap());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bool render_status;
        for(int i=0; i<RENDER_BATCH_COLUMNS; i++)
            render_status = engine->renderLoop(pWorld); // RenderLoop() raytraces 1 column of pixels at a time.
        if(!render_status)
        {
            // Update texture
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texImage);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, camera->getBitmap());
        }

        ImGui::Begin("Photon Mapping", NULL, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Size: %d x %d", image_width, image_height);
        if(ImGui::Button("Save")){
          char filename[] = "img.png";
          stbi_write_png(filename, image_width, image_height, 3, camera->getBitmap(),0);
        }
        //Display render view - fit to width
        int win_w, win_h;
        glfwGetWindowSize(window, &win_w, &win_h);
        float image_aspect = (float)image_width/(float)image_height;
        float frac = 0.95; // ensure no horizontal scrolling
        ImGui::Image((void*)(intptr_t)texImage, ImVec2(frac*win_w, frac*win_w/image_aspect), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteTextures(1, &texImage);

    cleanup(window);

    return 0;
}
