(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot213 shot247 shot336 - shot
      ingredient37 ingredient191 ingredient452 ingredient25 - ingredient
      cocktail1 - cocktail
      dispenser447 dispenser389 dispenser63 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot213)
  (ontable shot247)
  (ontable shot336)
  (dispenses dispenser447 ingredient37)
  (dispenses dispenser389 ingredient191)
  (dispenses dispenser63 ingredient452)
  (dispenses dispenser377 ingredient25)
  (clean shaker168)
  (clean shot213)
  (clean shot247)
  (clean shot336)
  (empty shaker168)
  (empty shot213)
  (empty shot247)
  (empty shot336)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient452)
  (cocktail-part2 cocktail1 ingredient191)
)
 (:goal
  (and
      (contains shot213 cocktail1)
      (contains shot247 cocktail1)
)))
