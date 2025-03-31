(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot344 shot437 shot114 - shot
      ingredient319 ingredient434 ingredient19 ingredient316 - ingredient
      cocktail216 - cocktail
      dispenser241 dispenser446 dispenser243 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot344)
  (ontable shot437)
  (ontable shot114)
  (dispenses dispenser241 ingredient319)
  (dispenses dispenser446 ingredient434)
  (dispenses dispenser243 ingredient19)
  (dispenses dispenser492 ingredient316)
  (clean shaker55)
  (clean shot344)
  (clean shot437)
  (clean shot114)
  (empty shaker55)
  (empty shot344)
  (empty shot437)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient319)
  (cocktail-part2 cocktail216 ingredient316)
)
 (:goal
  (and
      (contains shot344 cocktail216)
      (contains shot437 ingredient19)
)))
