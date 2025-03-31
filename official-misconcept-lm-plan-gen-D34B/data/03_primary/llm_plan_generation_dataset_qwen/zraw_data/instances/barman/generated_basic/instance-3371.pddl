(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot242 shot199 shot294 - shot
      ingredient497 ingredient246 ingredient498 - ingredient
      cocktail187 - cocktail
      dispenser38 dispenser229 dispenser416 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot242)
  (ontable shot199)
  (ontable shot294)
  (dispenses dispenser38 ingredient497)
  (dispenses dispenser229 ingredient246)
  (dispenses dispenser416 ingredient498)
  (clean shaker15)
  (clean shot242)
  (clean shot199)
  (clean shot294)
  (empty shaker15)
  (empty shot242)
  (empty shot199)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail187 ingredient246)
  (cocktail-part2 cocktail187 ingredient497)
)
 (:goal
  (and
      (contains shot242 cocktail187)
      (contains shot199 cocktail187)
)))
