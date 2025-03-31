; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 485096

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child160 - child
    bread459 - bread-portion
    content82 - content-portion
    tray116 tray33 tray150 - tray
    table167 table118 table392 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray116 kitchen)
     (at tray33 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_content content82)
     (not_allergic_gluten child160)
     (waiting child160 table118)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child160)
    )
  )
)
