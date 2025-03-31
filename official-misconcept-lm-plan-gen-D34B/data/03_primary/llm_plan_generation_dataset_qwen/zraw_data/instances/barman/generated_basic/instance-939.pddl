(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot208 shot453 shot251 - shot
      ingredient443 ingredient261 ingredient296 ingredient276 - ingredient
      cocktail1 - cocktail
      dispenser40 dispenser241 dispenser374 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot208)
  (ontable shot453)
  (ontable shot251)
  (dispenses dispenser40 ingredient443)
  (dispenses dispenser241 ingredient261)
  (dispenses dispenser374 ingredient296)
  (dispenses dispenser239 ingredient276)
  (clean shaker449)
  (clean shot208)
  (clean shot453)
  (clean shot251)
  (empty shaker449)
  (empty shot208)
  (empty shot453)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient443)
  (cocktail-part2 cocktail1 ingredient296)
)
 (:goal
  (and
      (contains shot208 cocktail1)
      (contains shot453 cocktail1)
)))
