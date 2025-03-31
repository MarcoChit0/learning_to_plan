; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 12923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 child457 - child
    bread212 bread16 - bread-portion
    content338 content203 - content-portion
    tray465 tray215 - tray
    table40 table84 table55 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray465 kitchen)
     (at tray215 kitchen)
     (at_kitchen_bread bread212)
     (at_kitchen_bread bread16)
     (at_kitchen_content content338)
     (at_kitchen_content content203)
     (not_allergic_gluten child134)
     (not_allergic_gluten child457)
     (waiting child134 table40)
     (waiting child457 table55)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child134)
     (served child457)
    )
  )
)
