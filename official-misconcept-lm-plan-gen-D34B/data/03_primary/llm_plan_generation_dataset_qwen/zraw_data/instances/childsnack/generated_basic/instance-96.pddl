; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 143661

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 child57 - child
    bread235 bread401 - bread-portion
    content490 content226 - content-portion
    tray75 tray439 - tray
    table179 table276 table59 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray75 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread235)
     (at_kitchen_bread bread401)
     (at_kitchen_content content490)
     (at_kitchen_content content226)
     (not_allergic_gluten child359)
     (not_allergic_gluten child57)
     (waiting child359 table276)
     (waiting child57 table59)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child359)
     (served child57)
    )
  )
)
