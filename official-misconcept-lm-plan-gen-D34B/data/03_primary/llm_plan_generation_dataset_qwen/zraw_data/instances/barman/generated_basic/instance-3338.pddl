(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot16 shot146 - shot
      ingredient134 ingredient355 - ingredient
      cocktail140 - cocktail
      dispenser180 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot16)
  (ontable shot146)
  (dispenses dispenser180 ingredient134)
  (dispenses dispenser207 ingredient355)
  (clean shaker234)
  (clean shot16)
  (clean shot146)
  (empty shaker234)
  (empty shot16)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient355)
  (cocktail-part2 cocktail140 ingredient134)
)
 (:goal
  (and
      (contains shot16 cocktail140)
)))
