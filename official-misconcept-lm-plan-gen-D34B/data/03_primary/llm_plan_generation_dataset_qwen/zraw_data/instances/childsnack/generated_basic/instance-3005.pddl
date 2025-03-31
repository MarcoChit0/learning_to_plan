; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 371021

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 child101 - child
    bread473 bread67 - bread-portion
    content132 content246 - content-portion
    tray229 tray7 tray152 tray193 - tray
    table370 table498 table323 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at tray7 kitchen)
     (at tray152 kitchen)
     (at tray193 kitchen)
     (at_kitchen_bread bread473)
     (at_kitchen_bread bread67)
     (at_kitchen_content content132)
     (at_kitchen_content content246)
     (not_allergic_gluten child101)
     (not_allergic_gluten child37)
     (waiting child37 table498)
     (waiting child101 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child37)
     (served child101)
    )
  )
)
