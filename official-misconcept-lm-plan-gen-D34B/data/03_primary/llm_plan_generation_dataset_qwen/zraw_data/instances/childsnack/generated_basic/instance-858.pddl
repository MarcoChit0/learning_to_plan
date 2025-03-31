; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 758458

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child117 - child
    bread144 bread117 - bread-portion
    content89 content219 - content-portion
    tray20 tray244 - tray
    table163 table338 table92 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread117)
     (at_kitchen_content content89)
     (at_kitchen_content content219)
     (no_gluten_bread bread117)
     (no_gluten_content content89)
     (allergic_gluten child481)
     (not_allergic_gluten child117)
     (waiting child481 table163)
     (waiting child117 table338)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child481)
     (served child117)
    )
  )
)
