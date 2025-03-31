; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 18973

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child58 child91 child30 - child
    bread60 bread193 bread142 - bread-portion
    content250 content420 content270 - content-portion
    tray433 tray459 tray155 - tray
    table203 table212 table226 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray433 kitchen)
     (at tray459 kitchen)
     (at tray155 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread142)
     (at_kitchen_content content250)
     (at_kitchen_content content420)
     (at_kitchen_content content270)
     (no_gluten_bread bread142)
     (no_gluten_content content250)
     (allergic_gluten child30)
     (not_allergic_gluten child91)
     (not_allergic_gluten child58)
     (waiting child58 table226)
     (waiting child91 table226)
     (waiting child30 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child58)
     (served child91)
     (served child30)
    )
  )
)
