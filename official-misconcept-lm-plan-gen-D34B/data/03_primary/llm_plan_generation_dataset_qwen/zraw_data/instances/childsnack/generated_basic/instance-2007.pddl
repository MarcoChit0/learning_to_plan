; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 500323

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child226 child151 - child
    bread113 bread169 - bread-portion
    content365 content5 - content-portion
    tray245 tray79 tray485 - tray
    table286 table264 table432 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray79 kitchen)
     (at tray485 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread169)
     (at_kitchen_content content365)
     (at_kitchen_content content5)
     (not_allergic_gluten child226)
     (not_allergic_gluten child151)
     (waiting child226 table432)
     (waiting child151 table432)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child226)
     (served child151)
    )
  )
)
