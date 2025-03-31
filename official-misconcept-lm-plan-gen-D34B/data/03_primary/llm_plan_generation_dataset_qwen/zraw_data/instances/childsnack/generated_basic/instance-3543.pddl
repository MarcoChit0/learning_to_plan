; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 891927

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child207 child158 - child
    bread48 bread383 - bread-portion
    content183 content207 - content-portion
    tray6 tray9 - tray
    table107 table332 table35 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread48)
     (at_kitchen_bread bread383)
     (at_kitchen_content content183)
     (at_kitchen_content content207)
     (not_allergic_gluten child158)
     (not_allergic_gluten child207)
     (waiting child207 table107)
     (waiting child158 table107)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child207)
     (served child158)
    )
  )
)
