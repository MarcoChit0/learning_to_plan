(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot353 shot110 shot322 - shot
      ingredient53 ingredient117 ingredient441 - ingredient
      cocktail12 - cocktail
      dispenser470 dispenser187 dispenser304 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot353)
  (ontable shot110)
  (ontable shot322)
  (dispenses dispenser470 ingredient53)
  (dispenses dispenser187 ingredient117)
  (dispenses dispenser304 ingredient441)
  (clean shaker189)
  (clean shot353)
  (clean shot110)
  (clean shot322)
  (empty shaker189)
  (empty shot353)
  (empty shot110)
  (empty shot322)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient441)
  (cocktail-part2 cocktail12 ingredient117)
)
 (:goal
  (and
      (contains shot353 cocktail12)
      (contains shot110 cocktail12)
)))
