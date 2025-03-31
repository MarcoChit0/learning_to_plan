; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 204482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child392 child196 - child
    bread316 bread438 - bread-portion
    content269 content434 - content-portion
    tray241 tray439 - tray
    table400 table152 table21 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_bread bread438)
     (at_kitchen_content content269)
     (at_kitchen_content content434)
     (not_allergic_gluten child392)
     (not_allergic_gluten child196)
     (waiting child392 table152)
     (waiting child196 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child392)
     (served child196)
    )
  )
)
