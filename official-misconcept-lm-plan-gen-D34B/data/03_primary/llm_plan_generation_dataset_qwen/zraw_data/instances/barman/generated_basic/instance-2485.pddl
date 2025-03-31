(define (problem prob)
 (:domain barman)
 (:objects 
      shaker131 - shaker
      left right - hand
      shot207 shot89 shot241 - shot
      ingredient323 ingredient202 ingredient176 - ingredient
      cocktail485 - cocktail
      dispenser42 dispenser214 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker131)
  (ontable shot207)
  (ontable shot89)
  (ontable shot241)
  (dispenses dispenser42 ingredient323)
  (dispenses dispenser214 ingredient202)
  (dispenses dispenser492 ingredient176)
  (clean shaker131)
  (clean shot207)
  (clean shot89)
  (clean shot241)
  (empty shaker131)
  (empty shot207)
  (empty shot89)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker131 l0)
  (shaker-level shaker131 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient176)
  (cocktail-part2 cocktail485 ingredient202)
)
 (:goal
  (and
      (contains shot207 cocktail485)
      (contains shot89 cocktail485)
)))
