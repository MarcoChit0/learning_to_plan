; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 117164

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 child114 - child
    bread180 bread347 - bread-portion
    content251 content203 - content-portion
    tray92 tray12 tray69 tray160 - tray
    table100 table236 table30 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray92 kitchen)
     (at tray12 kitchen)
     (at tray69 kitchen)
     (at tray160 kitchen)
     (at_kitchen_bread bread180)
     (at_kitchen_bread bread347)
     (at_kitchen_content content251)
     (at_kitchen_content content203)
     (no_gluten_bread bread180)
     (no_gluten_content content203)
     (allergic_gluten child152)
     (not_allergic_gluten child114)
     (waiting child152 table30)
     (waiting child114 table100)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child152)
     (served child114)
    )
  )
)
