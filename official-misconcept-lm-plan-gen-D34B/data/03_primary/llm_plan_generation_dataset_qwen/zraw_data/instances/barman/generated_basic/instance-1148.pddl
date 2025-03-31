(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot466 shot411 - shot
      ingredient140 ingredient165 ingredient103 ingredient89 - ingredient
      cocktail1 - cocktail
      dispenser276 dispenser0 dispenser7 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot466)
  (ontable shot411)
  (dispenses dispenser276 ingredient140)
  (dispenses dispenser0 ingredient165)
  (dispenses dispenser7 ingredient103)
  (dispenses dispenser116 ingredient89)
  (clean shaker448)
  (clean shot466)
  (clean shot411)
  (empty shaker448)
  (empty shot466)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient89)
  (cocktail-part2 cocktail1 ingredient140)
)
 (:goal
  (and
      (contains shot466 cocktail1)
)))
