(define (problem prob)
 (:domain barman)
 (:objects 
      shaker444 - shaker
      left right - hand
      shot432 shot80 shot314 - shot
      ingredient111 ingredient39 ingredient86 - ingredient
      cocktail1 - cocktail
      dispenser43 dispenser210 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker444)
  (ontable shot432)
  (ontable shot80)
  (ontable shot314)
  (dispenses dispenser43 ingredient111)
  (dispenses dispenser210 ingredient39)
  (dispenses dispenser18 ingredient86)
  (clean shaker444)
  (clean shot432)
  (clean shot80)
  (clean shot314)
  (empty shaker444)
  (empty shot432)
  (empty shot80)
  (empty shot314)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker444 l0)
  (shaker-level shaker444 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient39)
  (cocktail-part2 cocktail1 ingredient86)
)
 (:goal
  (and
      (contains shot432 cocktail1)
      (contains shot80 cocktail1)
)))
