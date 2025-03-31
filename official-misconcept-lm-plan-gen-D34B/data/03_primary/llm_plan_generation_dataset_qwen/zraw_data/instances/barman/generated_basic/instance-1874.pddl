(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot97 shot466 - shot
      ingredient187 ingredient36 - ingredient
      cocktail245 - cocktail
      dispenser38 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot97)
  (ontable shot466)
  (dispenses dispenser38 ingredient187)
  (dispenses dispenser308 ingredient36)
  (clean shaker448)
  (clean shot97)
  (clean shot466)
  (empty shaker448)
  (empty shot97)
  (empty shot466)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient36)
  (cocktail-part2 cocktail245 ingredient187)
)
 (:goal
  (and
      (contains shot97 cocktail245)
)))
