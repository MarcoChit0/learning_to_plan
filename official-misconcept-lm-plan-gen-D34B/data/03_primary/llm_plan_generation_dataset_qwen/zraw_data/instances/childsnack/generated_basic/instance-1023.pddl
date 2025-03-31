; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 143624

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 child298 child459 - child
    bread7 bread216 bread320 - bread-portion
    content396 content10 content301 - content-portion
    tray200 - tray
    table130 table298 table140 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread216)
     (at_kitchen_bread bread320)
     (at_kitchen_content content396)
     (at_kitchen_content content10)
     (at_kitchen_content content301)
     (not_allergic_gluten child298)
     (not_allergic_gluten child459)
     (not_allergic_gluten child394)
     (waiting child394 table298)
     (waiting child298 table140)
     (waiting child459 table130)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child394)
     (served child298)
     (served child459)
    )
  )
)
