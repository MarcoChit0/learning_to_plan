(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot275 shot288 - shot
      ingredient185 ingredient114 ingredient3 ingredient423 - ingredient
      cocktail428 - cocktail
      dispenser270 dispenser471 dispenser366 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot275)
  (ontable shot288)
  (dispenses dispenser270 ingredient185)
  (dispenses dispenser471 ingredient114)
  (dispenses dispenser366 ingredient3)
  (dispenses dispenser207 ingredient423)
  (clean shaker109)
  (clean shot275)
  (clean shot288)
  (empty shaker109)
  (empty shot275)
  (empty shot288)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail428 ingredient423)
  (cocktail-part2 cocktail428 ingredient114)
)
 (:goal
  (and
      (contains shot275 cocktail428)
)))
