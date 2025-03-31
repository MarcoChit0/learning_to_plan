; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 633092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 child173 - child
    bread468 bread434 - bread-portion
    content106 content177 - content-portion
    tray235 tray87 tray407 tray473 - tray
    table22 table223 table158 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at tray87 kitchen)
     (at tray407 kitchen)
     (at tray473 kitchen)
     (at_kitchen_bread bread468)
     (at_kitchen_bread bread434)
     (at_kitchen_content content106)
     (at_kitchen_content content177)
     (not_allergic_gluten child231)
     (not_allergic_gluten child173)
     (waiting child231 table158)
     (waiting child173 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child231)
     (served child173)
    )
  )
)
