; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 786327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 child495 child194 - child
    bread448 bread350 bread113 - bread-portion
    content257 content27 content456 - content-portion
    tray333 - tray
    table418 table225 table127 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread350)
     (at_kitchen_bread bread113)
     (at_kitchen_content content257)
     (at_kitchen_content content27)
     (at_kitchen_content content456)
     (not_allergic_gluten child495)
     (not_allergic_gluten child65)
     (not_allergic_gluten child194)
     (waiting child65 table418)
     (waiting child495 table418)
     (waiting child194 table418)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child65)
     (served child495)
     (served child194)
    )
  )
)
