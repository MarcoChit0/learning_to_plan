; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 52940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 child205 - child
    bread121 bread203 - bread-portion
    content11 content208 - content-portion
    tray474 tray233 tray14 - tray
    table231 table15 table435 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray474 kitchen)
     (at tray233 kitchen)
     (at tray14 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_bread bread203)
     (at_kitchen_content content11)
     (at_kitchen_content content208)
     (no_gluten_bread bread203)
     (no_gluten_content content11)
     (allergic_gluten child205)
     (not_allergic_gluten child210)
     (waiting child210 table435)
     (waiting child205 table15)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child210)
     (served child205)
    )
  )
)
