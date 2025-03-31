; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 716623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child118 child469 - child
    bread199 bread275 - bread-portion
    content33 content94 - content-portion
    tray286 tray342 tray29 - tray
    table136 table40 table90 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at tray342 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread275)
     (at_kitchen_content content33)
     (at_kitchen_content content94)
     (no_gluten_bread bread199)
     (no_gluten_content content33)
     (allergic_gluten child469)
     (not_allergic_gluten child118)
     (waiting child118 table90)
     (waiting child469 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child118)
     (served child469)
    )
  )
)
