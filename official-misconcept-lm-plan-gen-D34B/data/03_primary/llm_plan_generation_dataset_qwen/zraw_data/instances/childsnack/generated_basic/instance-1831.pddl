; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 315430

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child40 - child
    bread7 bread87 - bread-portion
    content121 content416 - content-portion
    tray488 - tray
    table319 table175 table23 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread87)
     (at_kitchen_content content121)
     (at_kitchen_content content416)
     (no_gluten_bread bread87)
     (no_gluten_content content416)
     (allergic_gluten child40)
     (not_allergic_gluten child265)
     (waiting child265 table319)
     (waiting child40 table23)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child265)
     (served child40)
    )
  )
)
