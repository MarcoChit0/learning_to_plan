; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 18527

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child149 - child
    bread454 bread396 - bread-portion
    content307 content186 - content-portion
    tray312 tray177 tray482 tray145 - tray
    table128 table225 table76 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray312 kitchen)
     (at tray177 kitchen)
     (at tray482 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread396)
     (at_kitchen_content content307)
     (at_kitchen_content content186)
     (no_gluten_bread bread454)
     (no_gluten_content content186)
     (allergic_gluten child142)
     (not_allergic_gluten child149)
     (waiting child142 table76)
     (waiting child149 table225)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child142)
     (served child149)
    )
  )
)
