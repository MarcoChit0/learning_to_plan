; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 906415

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child498 child77 - child
    bread474 bread16 - bread-portion
    content135 content296 - content-portion
    tray191 - tray
    table116 table442 table158 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at_kitchen_bread bread474)
     (at_kitchen_bread bread16)
     (at_kitchen_content content135)
     (at_kitchen_content content296)
     (no_gluten_bread bread16)
     (no_gluten_content content296)
     (allergic_gluten child77)
     (not_allergic_gluten child498)
     (waiting child498 table442)
     (waiting child77 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child498)
     (served child77)
    )
  )
)
