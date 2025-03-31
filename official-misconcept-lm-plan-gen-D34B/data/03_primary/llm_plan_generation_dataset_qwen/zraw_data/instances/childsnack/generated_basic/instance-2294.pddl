; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 890211

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child200 - child
    bread36 bread77 - bread-portion
    content232 content265 - content-portion
    tray427 - tray
    table35 table331 table236 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray427 kitchen)
     (at_kitchen_bread bread36)
     (at_kitchen_bread bread77)
     (at_kitchen_content content232)
     (at_kitchen_content content265)
     (no_gluten_bread bread36)
     (no_gluten_content content232)
     (allergic_gluten child184)
     (not_allergic_gluten child200)
     (waiting child184 table35)
     (waiting child200 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child184)
     (served child200)
    )
  )
)
