; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 884706

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 child258 child75 - child
    bread163 bread246 bread446 - bread-portion
    content336 content133 content260 - content-portion
    tray212 - tray
    table200 table172 table274 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at_kitchen_bread bread163)
     (at_kitchen_bread bread246)
     (at_kitchen_bread bread446)
     (at_kitchen_content content336)
     (at_kitchen_content content133)
     (at_kitchen_content content260)
     (not_allergic_gluten child258)
     (not_allergic_gluten child174)
     (not_allergic_gluten child75)
     (waiting child174 table172)
     (waiting child258 table172)
     (waiting child75 table200)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child174)
     (served child258)
     (served child75)
    )
  )
)
