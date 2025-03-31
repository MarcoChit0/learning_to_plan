; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 393837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child261 - child
    bread14 bread77 - bread-portion
    content31 content200 - content-portion
    tray433 tray25 - tray
    table224 table282 table136 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray433 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread77)
     (at_kitchen_content content31)
     (at_kitchen_content content200)
     (not_allergic_gluten child194)
     (not_allergic_gluten child261)
     (waiting child194 table136)
     (waiting child261 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child194)
     (served child261)
    )
  )
)
