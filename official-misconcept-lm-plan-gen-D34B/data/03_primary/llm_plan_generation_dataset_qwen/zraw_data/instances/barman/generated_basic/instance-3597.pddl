(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot97 shot319 shot307 shot302 - shot
      ingredient100 ingredient354 ingredient39 - ingredient
      cocktail293 - cocktail
      dispenser288 dispenser313 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot97)
  (ontable shot319)
  (ontable shot307)
  (ontable shot302)
  (dispenses dispenser288 ingredient100)
  (dispenses dispenser313 ingredient354)
  (dispenses dispenser310 ingredient39)
  (clean shaker321)
  (clean shot97)
  (clean shot319)
  (clean shot307)
  (clean shot302)
  (empty shaker321)
  (empty shot97)
  (empty shot319)
  (empty shot307)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient100)
  (cocktail-part2 cocktail293 ingredient39)
)
 (:goal
  (and
      (contains shot97 cocktail293)
      (contains shot319 ingredient39)
      (contains shot307 cocktail293)
)))
