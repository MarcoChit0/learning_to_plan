; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 168097

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child273 - child
    bread183 bread369 - bread-portion
    content411 content447 - content-portion
    tray372 tray422 - tray
    table279 table378 table495 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray372 kitchen)
     (at tray422 kitchen)
     (at_kitchen_bread bread183)
     (at_kitchen_bread bread369)
     (at_kitchen_content content411)
     (at_kitchen_content content447)
     (not_allergic_gluten child273)
     (not_allergic_gluten child10)
     (waiting child10 table378)
     (waiting child273 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child273)
    )
  )
)
