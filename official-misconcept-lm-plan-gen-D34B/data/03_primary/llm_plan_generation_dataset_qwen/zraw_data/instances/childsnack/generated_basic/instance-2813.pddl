; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 159981

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child77 child436 - child
    bread209 bread473 - bread-portion
    content172 content267 - content-portion
    tray105 tray213 - tray
    table59 table480 table160 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread209)
     (at_kitchen_bread bread473)
     (at_kitchen_content content172)
     (at_kitchen_content content267)
     (not_allergic_gluten child436)
     (not_allergic_gluten child77)
     (waiting child77 table480)
     (waiting child436 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child77)
     (served child436)
    )
  )
)
