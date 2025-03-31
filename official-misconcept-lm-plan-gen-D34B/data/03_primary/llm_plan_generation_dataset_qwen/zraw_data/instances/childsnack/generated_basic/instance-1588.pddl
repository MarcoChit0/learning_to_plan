; child-snack task with 3 children and 0.6 gluten factor 
; constant factor of 1.7
; random seed: 468324

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child478 child255 - child
    bread44 bread477 bread249 - bread-portion
    content463 content32 content286 - content-portion
    tray158 tray498 - tray
    table167 table487 table232 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray158 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_bread bread477)
     (at_kitchen_bread bread249)
     (at_kitchen_content content463)
     (at_kitchen_content content32)
     (at_kitchen_content content286)
     (no_gluten_bread bread44)
     (no_gluten_content content463)
     (allergic_gluten child479)
     (not_allergic_gluten child478)
     (not_allergic_gluten child255)
     (waiting child479 table167)
     (waiting child478 table232)
     (waiting child255 table232)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child479)
     (served child478)
     (served child255)
    )
  )
)
