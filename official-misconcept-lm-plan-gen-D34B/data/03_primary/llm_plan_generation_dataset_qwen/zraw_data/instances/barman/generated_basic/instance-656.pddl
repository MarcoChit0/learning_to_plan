(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot353 shot265 - shot
      ingredient458 ingredient58 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser137 dispenser121 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot353)
  (ontable shot265)
  (dispenses dispenser137 ingredient458)
  (dispenses dispenser121 ingredient58)
  (dispenses dispenser341 ingredient250)
  (clean shaker48)
  (clean shot353)
  (clean shot265)
  (empty shaker48)
  (empty shot353)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient458)
  (cocktail-part2 cocktail1 ingredient58)
)
 (:goal
  (and
      (contains shot353 cocktail1)
)))
