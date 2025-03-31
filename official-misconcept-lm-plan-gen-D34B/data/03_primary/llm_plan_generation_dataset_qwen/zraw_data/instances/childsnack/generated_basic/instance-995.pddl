; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 501009

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child151 child262 - child
    bread441 bread158 bread220 - bread-portion
    content74 content290 content458 - content-portion
    tray25 - tray
    table452 table23 table275 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray25 kitchen)
     (at_kitchen_bread bread441)
     (at_kitchen_bread bread158)
     (at_kitchen_bread bread220)
     (at_kitchen_content content74)
     (at_kitchen_content content290)
     (at_kitchen_content content458)
     (not_allergic_gluten child262)
     (not_allergic_gluten child151)
     (not_allergic_gluten child73)
     (waiting child73 table23)
     (waiting child151 table452)
     (waiting child262 table23)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child73)
     (served child151)
     (served child262)
    )
  )
)
