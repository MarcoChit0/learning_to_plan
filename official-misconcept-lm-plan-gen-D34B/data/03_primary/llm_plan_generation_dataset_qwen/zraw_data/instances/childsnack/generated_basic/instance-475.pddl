; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 269359

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 child456 - child
    bread459 bread195 - bread-portion
    content497 content419 - content-portion
    tray187 tray232 - tray
    table213 table42 table72 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray187 kitchen)
     (at tray232 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread195)
     (at_kitchen_content content497)
     (at_kitchen_content content419)
     (not_allergic_gluten child176)
     (not_allergic_gluten child456)
     (waiting child176 table72)
     (waiting child456 table72)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child176)
     (served child456)
    )
  )
)
