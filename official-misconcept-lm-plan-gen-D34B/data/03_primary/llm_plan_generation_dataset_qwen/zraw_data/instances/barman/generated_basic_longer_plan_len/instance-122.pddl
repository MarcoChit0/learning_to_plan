(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot25 shot249 shot170 - shot
      ingredient96 ingredient114 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser297 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot25)
  (ontable shot249)
  (ontable shot170)
  (dispenses dispenser297 ingredient96)
  (dispenses dispenser175 ingredient114)
  (clean shaker139)
  (clean shot25)
  (clean shot249)
  (clean shot170)
  (empty shaker139)
  (empty shot25)
  (empty shot249)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient114)
  (cocktail-part2 cocktail1 ingredient96)
  (cocktail-part1 cocktail2 ingredient96)
  (cocktail-part2 cocktail2 ingredient114)
)
 (:goal
  (and
      (contains shot25 cocktail1)
      (contains shot249 cocktail2)
)))
