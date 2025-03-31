(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot356 shot75 - shot
      ingredient369 ingredient393 ingredient359 - ingredient
      cocktail485 - cocktail
      dispenser277 dispenser178 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot356)
  (ontable shot75)
  (dispenses dispenser277 ingredient369)
  (dispenses dispenser178 ingredient393)
  (dispenses dispenser484 ingredient359)
  (clean shaker48)
  (clean shot356)
  (clean shot75)
  (empty shaker48)
  (empty shot356)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient393)
  (cocktail-part2 cocktail485 ingredient359)
)
 (:goal
  (and
      (contains shot356 cocktail485)
)))
