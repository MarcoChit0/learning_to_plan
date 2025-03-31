; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 962678

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child213 child20 - child
    bread206 bread225 - bread-portion
    content247 content47 - content-portion
    tray71 tray118 tray184 - tray
    table400 table64 table164 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray71 kitchen)
     (at tray118 kitchen)
     (at tray184 kitchen)
     (at_kitchen_bread bread206)
     (at_kitchen_bread bread225)
     (at_kitchen_content content247)
     (at_kitchen_content content47)
     (no_gluten_bread bread206)
     (no_gluten_content content247)
     (allergic_gluten child20)
     (not_allergic_gluten child213)
     (waiting child213 table400)
     (waiting child20 table400)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child213)
     (served child20)
    )
  )
)
