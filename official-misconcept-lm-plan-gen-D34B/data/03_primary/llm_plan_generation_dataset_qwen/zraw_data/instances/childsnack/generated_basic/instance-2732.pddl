; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 281431

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 child134 child45 - child
    bread488 bread404 bread446 - bread-portion
    content133 content281 content348 - content-portion
    tray469 tray164 tray376 - tray
    table220 table346 table347 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at tray164 kitchen)
     (at tray376 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_bread bread404)
     (at_kitchen_bread bread446)
     (at_kitchen_content content133)
     (at_kitchen_content content281)
     (at_kitchen_content content348)
     (not_allergic_gluten child134)
     (not_allergic_gluten child45)
     (not_allergic_gluten child256)
     (waiting child256 table346)
     (waiting child134 table347)
     (waiting child45 table220)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child256)
     (served child134)
     (served child45)
    )
  )
)
