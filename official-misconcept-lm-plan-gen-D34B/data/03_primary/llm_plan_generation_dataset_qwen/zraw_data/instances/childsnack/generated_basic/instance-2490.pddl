; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 549023

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child160 child476 - child
    bread383 bread210 - bread-portion
    content289 content151 - content-portion
    tray360 tray220 - tray
    table234 table435 table476 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_bread bread210)
     (at_kitchen_content content289)
     (at_kitchen_content content151)
     (not_allergic_gluten child160)
     (not_allergic_gluten child476)
     (waiting child160 table435)
     (waiting child476 table234)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child160)
     (served child476)
    )
  )
)
