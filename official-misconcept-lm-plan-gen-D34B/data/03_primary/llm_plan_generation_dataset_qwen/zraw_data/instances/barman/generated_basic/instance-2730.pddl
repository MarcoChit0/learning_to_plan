(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot260 shot167 shot270 - shot
      ingredient377 ingredient347 ingredient391 - ingredient
      cocktail305 - cocktail
      dispenser224 dispenser231 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot260)
  (ontable shot167)
  (ontable shot270)
  (dispenses dispenser224 ingredient377)
  (dispenses dispenser231 ingredient347)
  (dispenses dispenser261 ingredient391)
  (clean shaker200)
  (clean shot260)
  (clean shot167)
  (clean shot270)
  (empty shaker200)
  (empty shot260)
  (empty shot167)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient347)
  (cocktail-part2 cocktail305 ingredient377)
)
 (:goal
  (and
      (contains shot260 cocktail305)
      (contains shot167 cocktail305)
)))
