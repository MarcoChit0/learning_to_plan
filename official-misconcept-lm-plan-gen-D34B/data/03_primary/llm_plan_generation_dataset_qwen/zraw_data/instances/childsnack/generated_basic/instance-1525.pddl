; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 237545

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 child76 - child
    bread163 bread379 - bread-portion
    content442 content286 - content-portion
    tray493 tray374 tray87 - tray
    table416 table488 table302 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at tray374 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread163)
     (at_kitchen_bread bread379)
     (at_kitchen_content content442)
     (at_kitchen_content content286)
     (no_gluten_bread bread379)
     (no_gluten_content content286)
     (allergic_gluten child68)
     (not_allergic_gluten child76)
     (waiting child68 table488)
     (waiting child76 table488)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child68)
     (served child76)
    )
  )
)
