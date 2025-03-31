; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 235395

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 - child
    bread299 - bread-portion
    content9 - content-portion
    tray358 tray326 tray170 tray330 - tray
    table3 table275 table207 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray326 kitchen)
     (at tray170 kitchen)
     (at tray330 kitchen)
     (at_kitchen_bread bread299)
     (at_kitchen_content content9)
     (not_allergic_gluten child407)
     (waiting child407 table207)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child407)
    )
  )
)
