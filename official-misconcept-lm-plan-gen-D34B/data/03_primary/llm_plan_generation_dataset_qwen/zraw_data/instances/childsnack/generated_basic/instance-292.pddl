; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 797928

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child83 - child
    bread139 - bread-portion
    content322 - content-portion
    tray211 - tray
    table30 table185 table311 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray211 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_content content322)
     (not_allergic_gluten child83)
     (waiting child83 table311)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child83)
    )
  )
)
