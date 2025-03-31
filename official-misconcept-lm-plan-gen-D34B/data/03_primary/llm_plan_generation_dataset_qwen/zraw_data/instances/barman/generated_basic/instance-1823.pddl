(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot4 shot383 shot418 - shot
      ingredient356 ingredient54 ingredient303 ingredient400 - ingredient
      cocktail226 - cocktail
      dispenser376 dispenser26 dispenser445 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot4)
  (ontable shot383)
  (ontable shot418)
  (dispenses dispenser376 ingredient356)
  (dispenses dispenser26 ingredient54)
  (dispenses dispenser445 ingredient303)
  (dispenses dispenser481 ingredient400)
  (clean shaker229)
  (clean shot4)
  (clean shot383)
  (clean shot418)
  (empty shaker229)
  (empty shot4)
  (empty shot383)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail226 ingredient356)
  (cocktail-part2 cocktail226 ingredient54)
)
 (:goal
  (and
      (contains shot4 cocktail226)
      (contains shot383 ingredient303)
)))
