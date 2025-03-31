(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot282 shot183 - shot
      ingredient35 ingredient336 ingredient458 ingredient441 - ingredient
      cocktail482 - cocktail
      dispenser382 dispenser10 dispenser102 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot282)
  (ontable shot183)
  (dispenses dispenser382 ingredient35)
  (dispenses dispenser10 ingredient336)
  (dispenses dispenser102 ingredient458)
  (dispenses dispenser446 ingredient441)
  (clean shaker445)
  (clean shot282)
  (clean shot183)
  (empty shaker445)
  (empty shot282)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient336)
  (cocktail-part2 cocktail482 ingredient35)
)
 (:goal
  (and
      (contains shot282 cocktail482)
)))
