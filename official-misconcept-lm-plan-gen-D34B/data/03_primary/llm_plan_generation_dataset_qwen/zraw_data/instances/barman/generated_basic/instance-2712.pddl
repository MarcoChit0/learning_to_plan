(define (problem prob)
 (:domain barman)
 (:objects 
      shaker176 - shaker
      left right - hand
      shot73 shot162 shot393 - shot
      ingredient312 ingredient30 - ingredient
      cocktail294 - cocktail
      dispenser191 dispenser169 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker176)
  (ontable shot73)
  (ontable shot162)
  (ontable shot393)
  (dispenses dispenser191 ingredient312)
  (dispenses dispenser169 ingredient30)
  (clean shaker176)
  (clean shot73)
  (clean shot162)
  (clean shot393)
  (empty shaker176)
  (empty shot73)
  (empty shot162)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker176 l0)
  (shaker-level shaker176 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail294 ingredient30)
  (cocktail-part2 cocktail294 ingredient312)
)
 (:goal
  (and
      (contains shot73 cocktail294)
      (contains shot162 ingredient30)
)))
