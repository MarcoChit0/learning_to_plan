; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 700941

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 child260 - child
    bread319 bread305 - bread-portion
    content226 content0 - content-portion
    tray185 tray408 tray394 tray237 - tray
    table156 table348 table211 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray408 kitchen)
     (at tray394 kitchen)
     (at tray237 kitchen)
     (at_kitchen_bread bread319)
     (at_kitchen_bread bread305)
     (at_kitchen_content content226)
     (at_kitchen_content content0)
     (no_gluten_bread bread319)
     (no_gluten_content content226)
     (allergic_gluten child260)
     (not_allergic_gluten child372)
     (waiting child372 table348)
     (waiting child260 table156)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child372)
     (served child260)
    )
  )
)
