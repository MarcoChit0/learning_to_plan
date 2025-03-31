(define (problem prob)
 (:domain barman)
 (:objects 
      shaker280 - shaker
      left right - hand
      shot191 shot260 shot463 - shot
      ingredient232 ingredient98 ingredient315 - ingredient
      cocktail159 - cocktail
      dispenser249 dispenser89 dispenser353 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker280)
  (ontable shot191)
  (ontable shot260)
  (ontable shot463)
  (dispenses dispenser249 ingredient232)
  (dispenses dispenser89 ingredient98)
  (dispenses dispenser353 ingredient315)
  (clean shaker280)
  (clean shot191)
  (clean shot260)
  (clean shot463)
  (empty shaker280)
  (empty shot191)
  (empty shot260)
  (empty shot463)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker280 l0)
  (shaker-level shaker280 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail159 ingredient232)
  (cocktail-part2 cocktail159 ingredient315)
)
 (:goal
  (and
      (contains shot191 cocktail159)
      (contains shot260 cocktail159)
)))
