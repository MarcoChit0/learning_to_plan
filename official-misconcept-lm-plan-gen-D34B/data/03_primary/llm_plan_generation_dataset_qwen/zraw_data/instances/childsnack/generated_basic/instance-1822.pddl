; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 164352

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child57 - child
    bread137 bread16 - bread-portion
    content480 content28 - content-portion
    tray210 - tray
    table331 table336 table479 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray210 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_bread bread16)
     (at_kitchen_content content480)
     (at_kitchen_content content28)
     (no_gluten_bread bread16)
     (no_gluten_content content28)
     (allergic_gluten child57)
     (not_allergic_gluten child144)
     (waiting child144 table336)
     (waiting child57 table331)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child144)
     (served child57)
    )
  )
)
